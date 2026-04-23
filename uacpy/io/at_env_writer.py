"""
Shared Acoustics Toolbox Environment File Writer

Provides unified ENV file writing for Kraken, SPARC, Scooter, and Field programs.
Eliminates code duplication across model implementations.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, TextIO, Tuple

from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.constants import (
    SSPType, BoundaryType, AttenuationUnits, VolumeAttenuation,
    parse_ssp_type, parse_boundary_type,
    C_LOW_FACTOR, C_HIGH_FACTOR,
)


class ATEnvWriter:
    """
    Unified writer for Acoustics Toolbox environment files

    Handles the complex .env file format used by:
    - Kraken (normal modes)
    - Scooter (FFP wavenumber integration)
    - SPARC (time-domain modes)
    - Field (field computation from modes)

    This centralizes the ENV file writing logic that was previously
    duplicated across 150+ lines in multiple model files.

    Features:
    ---------
    - SSP interpolation type mapping
    - Boundary condition formatting
    - Volume attenuation codes
    - SSP depth clipping (handles floating-point precision)
    - Range-dependent bottom support
    """

    @staticmethod
    def write_header(
        f: TextIO,
        env: Environment,
        source: Source,
        ssp_type: SSPType,
        surface_type: BoundaryType,
        attenuation_unit: AttenuationUnits = AttenuationUnits.DB_PER_WAVELENGTH,
        volume_attenuation: Optional[VolumeAttenuation] = None,
        frequencies: Optional[np.ndarray] = None,
        n_media_override: Optional[int] = None,
        topopt_extra: str = '',
    ) -> None:
        """
        Write header section (title, frequency, TopOpt)

        Parameters
        ----------
        f : TextIO
            Open file handle
        env : Environment
            Environment configuration
        source : Source
            Source configuration
        ssp_type : SSPType
            SSP interpolation type
        surface_type : BoundaryType
            Surface boundary condition
        attenuation_unit : AttenuationUnits
            Attenuation units (default: dB/wavelength)
        volume_attenuation : VolumeAttenuation, optional
            Volume attenuation formula
        frequencies : ndarray, optional
            Frequency vector for broadband runs. If provided, TopOpt(6) is set
            to 'B' and the frequency vector is written after TopOpt.
        n_media_override : int, optional
            Override NMedia value. Used by multi-profile writer to ensure
            all profiles have the same NMedia.
        topopt_extra : str, optional
            Extra characters appended to TopOpt beyond position 6 (e.g.
            Scooter's TopOpt(7:7)='0' to zero out stabilising attenuation —
            see ``scooter.f90:81``). Default: empty.
        """
        # Title
        f.write(f"'{env.name}'\n")

        # Frequency
        f.write(f"{source.frequency[0]:.6f}\n")

        # Number of media: 1 (water column) + N sediment layers if layered bottom
        if n_media_override is not None:
            n_media = n_media_override
        else:
            n_media = 1
            if hasattr(env, 'bottom_layered') and env.bottom_layered is not None:
                n_media += len(env.bottom_layered.layers)
        f.write(f"{n_media}\n")

        # TopOpt string (>=6 characters): '<ssp_code><surface_code><atten_unit><vol_atten> <broadband>'
        # plus any model-specific ``topopt_extra`` (e.g. Scooter TopOpt(7)).
        ssp_code = ssp_type.to_acoustics_toolbox_code()
        surface_code = surface_type.to_acoustics_toolbox_code()
        atten_code = attenuation_unit.to_char()
        vol_atten_code = volume_attenuation.to_char() if volume_attenuation else ' '

        broadband_code = 'B' if frequencies is not None and len(frequencies) > 1 else ' '

        topopt = f"{ssp_code}{surface_code}{atten_code}{vol_atten_code} {broadband_code}{topopt_extra}"
        f.write(f"'{topopt}'\n")

        # Write surface halfspace properties when top BC is 'A'
        from uacpy.io.env_writer import write_surface_halfspace
        write_surface_halfspace(f, env)

    @staticmethod
    def write_fg_params(f: TextIO, params: Tuple[float, float, float, float]) -> None:
        """
        Write Francois-Garrison volume attenuation parameters.

        The AT Fortran ``ReadTopOpt`` routine reads one record of
        ``T, Salinity, pH, z_bar`` immediately after the TopOpt line when
        ``TopOpt(4)='F'``.

        Parameters
        ----------
        f : TextIO
            Open file handle
        params : tuple of 4 floats
            (T, S, pH, z_bar): temperature (degC), salinity (psu), pH,
            mean depth (m).
        """
        if params is None or len(params) != 4:
            raise ValueError(
                "francois_garrison_params must be a 4-tuple (T, S, pH, z_bar)"
            )
        T, S, pH, z_bar = params
        f.write(f"{T:.4f} {S:.4f} {pH:.4f} {z_bar:.4f}\n")

    @staticmethod
    def write_bio_layers(f: TextIO, bio_layers) -> None:
        """
        Write biological attenuation layers.

        The AT Fortran ``ReadTopOpt`` routine reads one count line followed
        by ``NBioLayers`` records of ``Z1, Z2, f0, Q, a0`` when
        ``TopOpt(4)='B'``.

        Parameters
        ----------
        f : TextIO
            Open file handle
        bio_layers : list of 5-tuples
            [(Z1, Z2, f0, Q, a0), ...] per layer.
        """
        if not bio_layers:
            raise ValueError("bio_layers must be a non-empty list of 5-tuples")
        f.write(f"{len(bio_layers)}\n")
        for layer in bio_layers:
            if len(layer) != 5:
                raise ValueError(
                    "Each bio layer must be a 5-tuple (Z1, Z2, f0, Q, a0)"
                )
            Z1, Z2, f0, Q, a0 = layer
            f.write(f"{Z1:.4f} {Z2:.4f} {f0:.4f} {Q:.4f} {a0:.6f}\n")

    @staticmethod
    def write_broadband_freqs(f: TextIO, frequencies: np.ndarray) -> None:
        """
        Write broadband frequency vector.

        In the AT file format, this is read by ReadfreqVec AFTER source/receiver
        depths (not immediately after TopOpt). Call this at the correct position
        in your env file writer.

        Parameters
        ----------
        f : TextIO
            Open file handle
        frequencies : ndarray
            Frequency vector in Hz
        """
        f.write(f"{len(frequencies)}\n")
        freq_str = " ".join(f"{freq:.6f}" for freq in frequencies)
        f.write(f"{freq_str} /\n")

    @staticmethod
    def write_ssp_section(
        f: TextIO,
        env: Environment,
        bottom_depth: float,
        n_mesh: int = 0,
        roughness: float = 0.0
    ) -> None:
        """
        Write SSP section with proper depth clipping

        Handles the critical issue of SSP depths exceeding bottom depth
        due to floating-point precision. Ensures SSP is clipped to the
        rounded bottom depth that will be written to the file.

        Parameters
        ----------
        f : TextIO
            Open file handle
        env : Environment
            Environment with SSP data
        bottom_depth : float
            Bottom depth in meters
        n_mesh : int, optional
            Number of mesh points (0 = automatic)
        roughness : float, optional
            Surface/interface roughness in meters
        """
        # Write mesh parameters
        f.write(f"{n_mesh}  {roughness:.1f}  {bottom_depth:.1f}\n")

        # CRITICAL: Match precision of written bottom depth
        # Bottom depth is written with .1f, so clip SSP to that rounded value
        bottom_depth_rounded = float(f"{bottom_depth:.1f}")

        # Build SSP array with clipping and duplicate removal
        ssp_to_write = []
        prev_depth = None

        for depth, c in env.ssp_data:
            # Clip SSP depth to not exceed rounded bottom depth
            if depth > bottom_depth_rounded:
                depth = bottom_depth_rounded

            # Skip duplicate depths (after clipping)
            if prev_depth is not None and abs(depth - prev_depth) < 1e-6:
                continue

            ssp_to_write.append((depth, c))
            prev_depth = depth

        # Ensure we have SSP at exactly bottom depth
        if len(ssp_to_write) > 0 and ssp_to_write[-1][0] < bottom_depth_rounded:
            # Extend to bottom with last sound speed
            last_c = ssp_to_write[-1][1]
            ssp_to_write.append((bottom_depth_rounded, last_c))

        # Write SSP data (Kraken format: depth sound_speed /)
        for depth, c in ssp_to_write:
            if env.attenuation > 0:
                f.write(f"  {depth:.6f} {c:.6f} {env.attenuation:.6f} /\n")
            else:
                f.write(f"  {depth:.6f} {c:.6f} /\n")

    @staticmethod
    def write_layer_sections(
        f: TextIO,
        env: 'Environment',
        seafloor_depth: float,
        n_mesh: int = 0,
    ) -> float:
        """
        Write sediment layer SSP blocks for LayeredBottom (NMEDIA > 1).

        Each SedimentLayer becomes an additional medium in the AT format.
        Each medium block has: mesh params line, then isovelocity SSP entries.

        Parameters
        ----------
        f : TextIO
            Open file handle
        env : Environment
            Environment with bottom_layered set
        seafloor_depth : float
            Depth of the seafloor (bottom of water column)
        n_mesh : int, optional
            Number of mesh points per medium (0 = auto). For multi-profile
            runs, use a fixed value to keep NTotal consistent across profiles.

        Returns
        -------
        float
            Depth of the bottom of the last sediment layer
            (i.e., top of the half-space)
        """
        if not hasattr(env, 'bottom_layered') or env.bottom_layered is None:
            return seafloor_depth

        layered = env.bottom_layered
        # Round to .1f precision to match mesh line format — Kraken's parser
        # requires the last SSP depth to exactly match the mesh max depth.
        current_depth = float(f"{seafloor_depth:.1f}")

        for layer in layered.layers:
            top_depth = current_depth
            bottom_depth = float(f"{current_depth + layer.thickness:.1f}")

            # Mesh params: n_mesh, sigma=0, max_depth
            f.write(f"{n_mesh}  0.0  {bottom_depth:.1f}\n")

            # Isovelocity SSP within layer: top and bottom at same speed
            # Format: depth  cp  cs  rho  ap  as /
            alpha_s = getattr(layer, 'shear_attenuation', 0.0)
            f.write(f"  {top_depth:.1f} {layer.sound_speed:.6f} "
                    f"{layer.shear_speed:.1f} {layer.density:.2f} "
                    f"{layer.attenuation:.2f} {alpha_s:.2f} /\n")
            f.write(f"  {bottom_depth:.1f} {layer.sound_speed:.6f} "
                    f"{layer.shear_speed:.1f} {layer.density:.2f} "
                    f"{layer.attenuation:.2f} {alpha_s:.2f} /\n")

            current_depth = bottom_depth

        return current_depth

    @staticmethod
    def write_bottom_section(
        f: TextIO,
        env: Environment,
        bottom_type: Optional[BoundaryType] = None,
        cp_bottom: Optional[float] = None,
        cs_bottom: Optional[float] = None,
        rho_bottom: Optional[float] = None,
        alpha_bottom: Optional[float] = None,
        filepath: Optional[Path] = None,
        verbose: bool = False,
        halfspace_depth: Optional[float] = None,
    ) -> None:
        """
        Write bottom boundary section

        Parameters
        ----------
        f : TextIO
            Open file handle
        env : Environment
            Environment configuration
        bottom_type : BoundaryType, optional
            Bottom boundary type (uses env.bottom.acoustic_type if None)
        cp_bottom : float, optional
            Bottom compressional sound speed (uses env.bottom.sound_speed if None)
        cs_bottom : float, optional
            Bottom shear sound speed (uses env.bottom.shear_speed if None)
        rho_bottom : float, optional
            Bottom density (uses env.bottom.density if None)
        alpha_bottom : float, optional
            Bottom attenuation (uses env.bottom.attenuation if None)
        filepath : Path, optional
            Path to the ENV file being written (needed for copying .brc files)
        verbose : bool, optional
            Print verbose output
        """
        # Get bottom properties
        if bottom_type is None:
            bottom_type = parse_boundary_type(env.bottom.acoustic_type)

        cp = cp_bottom if cp_bottom is not None else env.bottom.sound_speed
        cs = cs_bottom if cs_bottom is not None else getattr(env.bottom, 'shear_speed', 0.0)
        rho = rho_bottom if rho_bottom is not None else env.bottom.density
        alpha = alpha_bottom if alpha_bottom is not None else env.bottom.attenuation

        # Write bottom boundary condition
        bottom_code = bottom_type.to_acoustics_toolbox_code()
        sigma = getattr(env.bottom, 'roughness', 0.0)  # Bottom roughness

        # Check for range-dependent bathymetry
        if hasattr(env, 'bathymetry') and len(env.bathymetry) > 1:
            # Range-dependent: append '~' to indicate .bty file
            f.write(f"'{bottom_code}~' {sigma:.1f}\n")
        else:
            # Range-independent
            f.write(f"'{bottom_code}' {sigma:.1f}\n")

        # Handle reflection coefficient file (type 'F')
        if bottom_code == 'F':
            # Copy reflection file to working directory
            if env.bottom.reflection_file and filepath is not None:
                import shutil
                brc_source = Path(env.bottom.reflection_file)
                if brc_source.exists():
                    # Copy to same directory as .env file with matching base name
                    brc_dest = filepath.with_suffix('.brc')
                    shutil.copy(brc_source, brc_dest)
                    if verbose:
                        print(f"[ATEnvWriter] Copied reflection file: {brc_source} -> {brc_dest}")
                else:
                    raise FileNotFoundError(
                        f"Reflection coefficient file not found: {env.bottom.reflection_file}\n"
                        f"Generate this file using BOUNCE or OASR models."
                    )
            elif env.bottom.reflection_file is None:
                raise ValueError(
                    "acoustic_type='file' requires reflection_file parameter.\n"
                    "Example: BoundaryProperties(acoustic_type='file', reflection_file='path/to/file.brc')"
                )

            # For 'F' type, write phase velocity bounds and rmax (not bottom properties)
            # These define the range of angles covered by the reflection coefficient table
            cmin = getattr(env.bottom, 'reflection_cmin', 1400.0)
            cmax = getattr(env.bottom, 'reflection_cmax', 10000.0)
            rmax_km = getattr(env.bottom, 'reflection_rmax_km', 10.0)
            f.write(f"{cmin:.2f}  {cmax:.2f}\n")
            f.write(f"{rmax_km:.2f}\n")

        # Write halfspace parameters (type 'A')
        elif bottom_code == 'A':  # Half-space
            if halfspace_depth is not None:
                z_bottom = halfspace_depth
            else:
                z_bottom = env.depth
                # If layered bottom, halfspace starts below all layers
                # Use env.depth (seafloor), not env.bottom.depth (may be stale)
                # Round to .1f to match layer section mesh line precision
                if hasattr(env, 'bottom_layered') and env.bottom_layered is not None:
                    z_bottom = float(f"{z_bottom:.1f}")
                    for layer in env.bottom_layered.layers:
                        z_bottom = float(f"{z_bottom + layer.thickness:.1f}")
            f.write(f"  {z_bottom:.2f}  {cp:.2f}  {cs:.1f}  "
                   f"{rho:.2f}  {alpha:.2f}  0.0 /\n")

    @staticmethod
    def write_source_depths(f: TextIO, source: Source) -> None:
        """Write source depth section"""
        n_sources = len(source.depth)
        f.write(f"{n_sources}\n")
        depths_str = " ".join([f"{d:.6f}" for d in source.depth])
        f.write(f"{depths_str} /\n")

    @staticmethod
    def write_receiver_depths(f: TextIO, receiver: Receiver) -> None:
        """Write receiver depth section"""
        n_rd = len(receiver.depths)
        f.write(f"{n_rd}\n")
        depths_str = " ".join([f"{d:.6f}" for d in receiver.depths])
        f.write(f"{depths_str} /\n")

    @staticmethod
    def write_receiver_ranges(f: TextIO, receiver: Receiver) -> None:
        """Write receiver range section"""
        n_rr = len(receiver.ranges)
        f.write(f"{n_rr}\n")
        # Convert to km
        ranges_str = " ".join([f"{r/1000.0:.6f}" for r in receiver.ranges])
        f.write(f"{ranges_str} /\n")

    @staticmethod
    def write_multi_profile_env(
        filepath: Path,
        segments: List[Tuple[float, 'Environment']],
        source: Source,
        receiver: Receiver,
        volume_attenuation: Optional[VolumeAttenuation] = None,
        **kwargs
    ) -> None:
        """
        Write multi-profile .env file for kraken.exe range-dependent mode.

        kraken.exe reads profile sections sequentially from a single .env file
        (via its ``Profile: DO iProf = 1, 9999`` loop), computing modes for
        each and writing them all into one .mod file.

        Each profile block contains: title, freq, NMedia, TopOpt, SSP,
        BotOpt, bottom halfspace, cLow/cHigh, RMax, source depths,
        receiver depths. Receiver ranges are NOT included (field.exe
        reads them from the .flp file).

        All profiles use the same n_mesh (fixed, non-zero) to ensure the
        .mod record length (``LRecordLength``) is consistent across
        profiles. kraken.exe sets the record length from the first
        profile and it must not increase for subsequent profiles
        (``krakenc.f90`` line 629). All profiles are also padded to the
        same NMedia so that NTotal (sum of mesh N across media) is
        identical for every profile.

        Parameters
        ----------
        filepath : Path
            Output .env file path
        segments : list of (range_km, Environment)
            Range segments. Each Environment must be range-independent.
        source : Source
            Source configuration (frequency, depth)
        receiver : Receiver
            Receiver configuration (depths for mode computation)
        volume_attenuation : VolumeAttenuation, optional
            Volume attenuation formula
        **kwargs
            n_mesh, roughness, c_low, c_high, rmax_km passed through
        """
        n_mesh = kwargs.get('n_mesh', 0)
        roughness = kwargs.get('roughness', 0.0)
        c_low = kwargs.get('c_low', None)
        c_high = kwargs.get('c_high', None)
        rmax_km = kwargs.get('rmax_km', 100.0)

        # Ensure n_mesh > 0 for consistency across profiles.
        # If caller didn't specify, compute from max depth.
        if n_mesh <= 0:
            freq = float(source.frequency[0])
            max_depth = max(seg.depth for _, seg in segments)
            n_mesh = max(500, int(max_depth * freq / 1500.0 * 20))

        # Determine max NMedia across all segments so every profile
        # can be padded to the same number of media (=> same NTotal).
        def _n_media(env_seg):
            n = 1  # water column
            if hasattr(env_seg, 'bottom_layered') and env_seg.bottom_layered is not None:
                n += len(env_seg.bottom_layered.layers)
            return n

        max_n_media = max(_n_media(seg) for _, seg in segments)

        # AT multi-profile kraken requires NMedia >= 2 for range-
        # dependent environments.  When all segments are NMedia=1
        # (water + halfspace only), we must add a sediment layer so
        # that total media depth can be held constant across profiles
        # even though the water column depth varies.  This matches the
        # AT convention (see tests/wedge: NMedia=2, constant total
        # depth for all 51 profiles).
        if max_n_media < 2:
            max_n_media = 2

        # Compute max total media depth across all profiles.
        # ALL profiles will be extended to this depth (constant total
        # depth) so that ReadSzRz doesn't clip receiver depths
        # differently per profile, which would change NzTab and break
        # the .mod record length.  This mirrors the AT convention
        # (see tests/wedge/runtests.m: total depth = 2000 m for all).
        def _total_depth(env_seg):
            d = env_seg.depth
            if hasattr(env_seg, 'bottom_layered') and env_seg.bottom_layered is not None:
                for layer in env_seg.bottom_layered.layers:
                    d += layer.thickness
            return d

        # Account for padding: each pad layer adds 0.1 m minimum,
        # so include the worst-case padding in the total depth.
        max_total_depth = max(
            _total_depth(seg) + 0.1 * (max_n_media - _n_media(seg))
            for _, seg in segments
        )

        # Build the layer list for each profile: water column + real
        # sediment layers + padding layers, totalling max_n_media media
        # and reaching max_total_depth.  This approach precomputes
        # everything before writing so that the last medium can be
        # extended to max_total_depth regardless of whether it's a real
        # or padding layer.
        max_total_rounded = float(f"{max_total_depth:.1f}")

        with open(filepath, 'w') as f:
            for i, (_range_km, env_seg) in enumerate(segments):
                ssp_type = parse_ssp_type(env_seg.ssp_type)
                # Respect the user's surface BC rather than silently
                # forcing every profile to VACUUM. segment_environment_by_range
                # copies env.surface onto each segment (see coupled_modes.py),
                # so env_seg.surface is always populated. Fall back to VACUUM
                # if a caller ever built a segment without one.
                surface_obj = getattr(env_seg, 'surface', None)
                if surface_obj is not None:
                    surface_type = parse_boundary_type(
                        surface_obj.acoustic_type
                    )
                else:
                    surface_type = BoundaryType.VACUUM
                bottom_type = parse_boundary_type(env_seg.bottom.acoustic_type)

                n_media_this = _n_media(env_seg)
                n_media_write = max_n_media

                ATEnvWriter.write_header(
                    f, env_seg, source,
                    ssp_type=ssp_type,
                    surface_type=surface_type,
                    volume_attenuation=volume_attenuation,
                    n_media_override=n_media_write
                )

                # --- Water column (medium 1) ---
                ATEnvWriter.write_ssp_section(
                    f, env_seg, env_seg.depth,
                    n_mesh=n_mesh,
                    roughness=roughness
                )

                # --- Sediment layers (media 2..n_media_this) ---
                # Collect real layers with their depths, then write
                # them together with any needed extensions.
                hs = env_seg.bottom
                seafloor = float(f"{env_seg.depth:.1f}")
                current_depth = seafloor
                real_layers = []

                if hasattr(env_seg, 'bottom_layered') and env_seg.bottom_layered is not None:
                    for layer in env_seg.bottom_layered.layers:
                        top = current_depth
                        bot = float(f"{current_depth + layer.thickness:.1f}")
                        real_layers.append((top, bot, layer))
                        current_depth = bot

                # Build full list of media 2..max_n_media
                # Real layers first, then padding up to max_n_media
                n_pad = n_media_write - n_media_this
                all_extra_media = []  # list of (top, bot, cp, cs, rho, ap, as_)

                for top, bot, layer in real_layers:
                    alpha_s = getattr(layer, 'shear_attenuation', 0.0)
                    all_extra_media.append(
                        (top, bot, layer.sound_speed, layer.shear_speed,
                         layer.density, layer.attenuation, alpha_s)
                    )

                # Add padding layers with halfspace properties
                # (must match halfspace cp, cs, rho, attenuation so the
                # padding-halfspace interface is acoustically transparent)
                hs_cs = getattr(hs, 'shear_speed', 0.0) or 0.0
                hs_as = getattr(hs, 'shear_attenuation', 0.0) or 0.0
                for _ in range(n_pad):
                    pad_top = current_depth
                    pad_bot = float(f"{current_depth + 0.1:.1f}")
                    all_extra_media.append(
                        (pad_top, pad_bot, hs.sound_speed, hs_cs,
                         hs.density, hs.attenuation, hs_as)
                    )
                    current_depth = pad_bot

                # Extend the LAST extra medium to max_total_depth so
                # that all profiles have the same total media depth.
                if all_extra_media:
                    last = all_extra_media[-1]
                    if last[1] < max_total_rounded:
                        all_extra_media[-1] = (
                            last[0], max_total_rounded,
                            last[2], last[3], last[4], last[5], last[6]
                        )

                # Write all extra media
                for top, bot, cp, cs, rho_v, ap, as_ in all_extra_media:
                    f.write(f"{n_mesh}  0.0  {bot:.1f}\n")
                    f.write(f"  {top:.1f} {cp:.6f} "
                            f"{cs:.1f} {rho_v:.2f} "
                            f"{ap:.2f} {as_:.2f} /\n")
                    f.write(f"  {bot:.1f} {cp:.6f} "
                            f"{cs:.1f} {rho_v:.2f} "
                            f"{ap:.2f} {as_:.2f} /\n")

                # Halfspace depth = bottom of all media
                if all_extra_media:
                    hs_depth = all_extra_media[-1][1]
                else:
                    hs_depth = seafloor

                ATEnvWriter.write_bottom_section(
                    f, env_seg, bottom_type=bottom_type,
                    filepath=filepath,
                    verbose=kwargs.get('verbose', False),
                    halfspace_depth=hs_depth,
                )

                # Phase speed limits (cLow, cHigh) — part of ReadEnvironment
                c_min = min(c for _, c in env_seg.ssp_data)
                c_max = max(
                    [c for _, c in env_seg.ssp_data] +
                    [env_seg.bottom.sound_speed]
                )
                _c_low = c_low if c_low is not None else c_min * C_LOW_FACTOR
                _c_high = c_high if c_high is not None else c_max * C_HIGH_FACTOR
                f.write(f"{_c_low:.1f} {_c_high:.1f}\n")

                # Maximum range (km) — part of ReadEnvironment
                f.write(f"{rmax_km:.1f}\n")

                ATEnvWriter.write_source_depths(f, source)
                ATEnvWriter.write_receiver_depths(f, receiver)


