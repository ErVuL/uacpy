"""
Acoustics Toolbox / OALIB environment-file writers.

Each function writes one logical block of the AT ``.env`` format onto an
open text handle, plus the ``.flp`` field-parameter writers used by
KrakenField. ``write_multi_profile_env`` and ``write_fieldflp`` /
``write_field3dflp`` write the full file.

Adoption across uacpy model wrappers:

- ``write_header``: Kraken, KrakenC, KrakenField, Scooter, Bounce.
  SPARC writes its own title/freq/NMedia line (`SPARC` has a 5th TopOpt
  position for ``output_mode``); Bellhop has its own writer entirely.
- ``write_bottom_section``: Kraken, KrakenC, KrakenField, Bounce.
  Scooter and SPARC open-code the bottom block because their ``'F'``
  (BRC) and ``'A'`` halfspace formats slightly differ.
- ``write_source_depths`` / ``write_receiver_depths`` /
  ``write_receiver_ranges``: every AT-family wrapper, including Bellhop.
- ``write_absorption_block`` (calls ``write_fg_params`` / ``write_bio_layers``):
  every AT-family wrapper including Bellhop. Drives output from
  ``env.absorption``.
- ``_BOUNDARY_TYPE_MAP`` / ``get_top_bc_code`` /
  ``write_surface_halfspace``: all AT-family wrappers including Bellhop.
"""

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union

from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.constants import (
    BoundaryType, AttenuationUnits,
    parse_boundary_type,
    C_LOW_FACTOR, C_HIGH_FACTOR,
)
from uacpy._log import log_message
from uacpy.io.utils import equally_spaced
from uacpy.io.units import m_to_km


_BOUNDARY_TYPE_MAP = {
    "vacuum": "V", "rigid": "R",
    "halfspace": "A", "half-space": "A",
    "file": "F", "precalc": "P",
    "grain-size": "G", "grain_size": "G", "grain": "G",
}


_AT_INTERP_TO_CODE = {
    'linear': 'C',
    'c-linear': 'C',
    'clin': 'C',
    'bilinear': 'C',
    'n2linear': 'N',
    'pchip': 'P',
    'cubic': 'S',
    'spline': 'S',
    'quad': 'Q',
    'analytic': 'A',
}


def resolve_ssp_interp(env: Environment, model_interp) -> str:
    """Return the user-facing ``interp_ssp`` value after auto-resolution.

    ``None`` means *auto*: pick ``'quad'`` when the env has a
    range-dependent SSP (matching Bellhop's ``.ssp`` quad-file path),
    otherwise ``'linear'``. Explicit values pass through unchanged.
    """
    if model_interp is None:
        return 'quad' if env.has_range_dependent_ssp() else 'linear'
    return str(model_interp).lower()


def resolve_ssp_topopt(env: Environment, model_interp) -> str:
    """Pick the AT ``TopOpt(1)`` character for an env / model pair.

    The model's ``interp_ssp`` (``None`` → auto / ``'linear'`` /
    ``'pchip'`` / ``'cubic'`` / ``'quad'`` / ``'n2linear'`` /
    ``'analytic'`` / …) drives the character via :data:`_AT_INTERP_TO_CODE`.
    The only env-side override is ``shape='isovelocity'`` which forces
    ``'C'`` (any connection scheme over constant data is constant). All
    other shape values (``'munk'``, ``'analytic'``, ``'n2linear'``,
    ``'measured'``) are informational — the model decides how to connect
    the samples.
    """
    shape = getattr(env.ssp, 'shape', 'measured')
    if shape == 'isovelocity':
        return 'C'
    key = resolve_ssp_interp(env, model_interp)
    if key not in _AT_INTERP_TO_CODE:
        raise ValueError(
            f"interp_ssp={model_interp!r} not recognised. Valid: "
            f"{sorted(set(_AT_INTERP_TO_CODE))} (or None for auto)"
        )
    return _AT_INTERP_TO_CODE[key]


def get_top_bc_code(env: Environment) -> str:
    """Return the single-character AT top boundary condition code."""
    return _BOUNDARY_TYPE_MAP.get(env.surface.acoustic_type.lower(), "V")


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

    # Validate range vector vs SSP matrix shape — each column of ``c``
    # is the profile at the corresponding range. Mismatched shapes will
    # otherwise produce a silently-malformed .ssp file that Bellhop
    # rejects deep in its run.
    if c.ndim != 2:
        raise ValueError(
            f"write_ssp: c must be 2-D (n_depth, n_ranges); got shape {c.shape}"
        )
    if c.shape[1] != Npts:
        raise ValueError(
            f"write_ssp: len(r_km) = {Npts} does not match c.shape[1] = "
            f"{c.shape[1]} (each column of c must be one profile)"
        )

    # AT/bellhopcuda's LDIFile reader treats each line as a separate
    # list-directed record (`LIST(SSPFile)` resets to the next line before
    # each read), so Npts and the range vector must live on different lines.
    with open(filepath, "w") as fid:
        fid.write(f"{Npts}\n")
        for r in r_km:
            fid.write(f"{r:6.3f}  ")
        fid.write("\n")
        # Sub-decimetre precision so Munk-style SSPs (e.g. 1502.345 m/s)
        # are not silently rounded; Acoustics-Toolbox parses free-format
        # so the extra digits are tolerated.
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                fid.write(f"{c[i, j]:8.4f} ")
            fid.write("\n")


def write_header(
    f: TextIO,
    env: Environment,
    source: Source,
    ssp_topopt: str,
    surface_type: BoundaryType,
    frequencies: Optional[np.ndarray] = None,
    n_media_override: Optional[int] = None,
    topopt_extra: str = '',
) -> None:
    """
    Write header section (title, frequency, TopOpt).

    TopOpt position 3 is hardwired to ``'W'`` (dB/wavelength) — uacpy's
    documented unit convention for every attenuation field. Position 4 is
    taken from ``env.absorption``: ``Thorp`` → ``'T'``,
    ``FrancoisGarrison`` → ``'F'``, ``Biological`` → ``'B'``,
    ``ConstantAbsorption`` or ``None`` → ``' '``. Per-formula follow-up
    lines (FG params, bio block) are emitted by
    :func:`write_absorption_block`, called separately by the caller
    immediately after this function.

    Parameters
    ----------
    f : TextIO
        Open file handle
    env : Environment
        Environment configuration (``env.absorption`` drives TopOpt(4))
    source : Source
        Source configuration
    ssp_topopt : str
        Pre-resolved single-character ``TopOpt(1)`` code (typically
        from :func:`resolve_ssp_topopt`).
    surface_type : BoundaryType
        Surface boundary condition
    frequencies : ndarray, optional
        Frequency vector for broadband runs. If provided, TopOpt(6) is set
        to ``'B'`` and the frequency vector is written after TopOpt.
    n_media_override : int, optional
        Override NMedia value. Used by multi-profile writer to ensure
        all profiles have the same NMedia.
    topopt_extra : str, optional
        Extra characters appended to TopOpt beyond position 6 (e.g.
        Scooter's TopOpt(7:7)='0' to zero out stabilising attenuation —
        see ``scooter.f90:81``). Default: empty.
    """
    f.write(f"'{env.name}'\n")
    f.write(f"{source.frequencies[0]:.6f}\n")

    if n_media_override is not None:
        n_media = n_media_override
    else:
        n_media = 1
        if env.has_layered_bottom():
            n_media += len(env.bottom.layers)
    f.write(f"{n_media}\n")

    ssp_code = ssp_topopt
    surface_code = surface_type.to_acoustics_toolbox_code()
    atten_code = AttenuationUnits.DB_PER_WAVELENGTH.to_char()
    vol_atten_code = (
        env.absorption.topopt_code() if env.absorption is not None else ' '
    )

    broadband_code = 'B' if frequencies is not None and len(frequencies) > 1 else ' '

    topopt = f"{ssp_code}{surface_code}{atten_code}{vol_atten_code} {broadband_code}{topopt_extra}"
    f.write(f"'{topopt}'\n")

    write_surface_halfspace(f, env)


def write_absorption_block(f: TextIO, env: Environment) -> None:
    """Emit the post-TopOpt absorption block (FG params or bio layers).

    For ``env.absorption`` of type :class:`FrancoisGarrison` writes one
    record ``T S pH z_bar``; for :class:`Biological` writes the layer
    count followed by one ``Z1 Z2 f0 Q a0`` record per layer. Other
    absorption types (Thorp, ConstantAbsorption, None) emit nothing.
    """
    from uacpy.core.absorption import Biological, FrancoisGarrison
    absorption = env.absorption
    if isinstance(absorption, FrancoisGarrison):
        write_fg_params(f, absorption.as_at_tuple())
    elif isinstance(absorption, Biological):
        write_bio_layers(f, absorption.as_at_tuples())


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
            "write_fg_params: params must be a 4-tuple (T, S, pH, z_bar)"
        )
    T, S, pH, z_bar = params
    f.write(f"{T:.4f} {S:.4f} {pH:.4f} {z_bar:.4f}\n")


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


def resolve_phase_speed_bounds(
    env: Environment,
    c_low: Optional[float] = None,
    c_high: Optional[float] = None,
) -> Tuple[float, float]:
    """Resolve effective ``(c_low, c_high)`` for an AT-family run.

    Precedence (same logic used by :func:`write_phase_speed_and_rmax`):
      1. Explicit caller values win.
      2. Otherwise: ``c_low = c_min · C_LOW_FACTOR`` and
         ``c_high = max(c_max, env.bottom.sound_speed) · C_HIGH_FACTOR``.

    Useful for model wrappers that want to log the resolved values
    before handing them to the writer.
    """
    if c_low is not None and c_high is not None:
        return float(c_low), float(c_high)
    ssp_pairs = env.ssp.to_pairs()
    c_min = float(ssp_pairs[:, 1].min())
    hs_c = env.halfspace_at_range(0.0).sound_speed
    c_max = max(float(ssp_pairs[:, 1].max()), float(hs_c))
    return (
        float(c_low) if c_low is not None else c_min * C_LOW_FACTOR,
        float(c_high) if c_high is not None else c_max * C_HIGH_FACTOR,
    )


def write_phase_speed_and_rmax(
    f: TextIO,
    env: Environment,
    *,
    rmax_m: float,
    c_low: Optional[float] = None,
    c_high: Optional[float] = None,
    rmax_format: str = "{:.1f}",
) -> None:
    """Write the cLow/cHigh phase-speed line and the RMax (km) line.

    cLow/cHigh resolve in this order:
      1. Explicit ``c_low`` / ``c_high`` (caller-supplied user override).
      2. SSP-derived: ``c_min·C_LOW_FACTOR`` and
         ``max(c_max, env.bottom.sound_speed)·C_HIGH_FACTOR``.

    ``rmax_m`` is converted to km. ``rmax_format`` controls the Fortran
    print width (Scooter/SPARC use ``"{:.6f}"`` to preserve sub-km
    precision; Kraken/KrakenField use the ``"{:.1f}"`` default).
    """
    _c_low, _c_high = resolve_phase_speed_bounds(env, c_low, c_high)
    f.write(f"{_c_low:.1f} {_c_high:.1f}\n")
    f.write(rmax_format.format(float(m_to_km(rmax_m))) + "\n")


def write_ssp_section(
    f: TextIO,
    env: Environment,
    bottom_depth: float,
    n_mesh: int = 0,
    roughness: float = 0.0
) -> None:
    """Write the SSP section with the deepest sample aligned to
    ``bottom_depth`` (rounded to the writer's ``.1f`` header precision).

    Both the header line and the SSP samples go through the same rounded
    depth so the AT parser sees ``ssp[-1].z == header.z_max`` exactly.
    Alignment delegates to :meth:`SoundSpeedProfile.extend_to`, which
    truncates with linear interpolation when the SSP runs past
    ``bottom_depth``, or extends by constant extrapolation when it falls
    short.
    """
    from uacpy.core.absorption import ConstantAbsorption
    bottom_depth_rounded = float(f"{bottom_depth:.1f}")
    f.write(f"{n_mesh}  {roughness:.1f}  {bottom_depth_rounded}\n")

    baseline = (
        env.absorption.value_db_per_wavelength
        if isinstance(env.absorption, ConstantAbsorption)
        else 0.0
    )
    for depth, c in env.ssp.extend_to(bottom_depth_rounded).to_pairs():
        if baseline > 0:
            f.write(f"  {depth:.6f} {c:.6f} {baseline:.6f} /\n")
        else:
            f.write(f"  {depth:.6f} {c:.6f} /\n")


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
        Environment with a LayeredBottom on env.bottom
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
    if not env.has_layered_bottom():
        return seafloor_depth

    layered = env.bottom
    # Round to .1f precision to match mesh line format — Kraken's parser
    # requires the last SSP depth to exactly match the mesh max depth.
    current_depth = float(f"{seafloor_depth:.1f}")

    for layer in layered.layers:
        top_depth = current_depth
        bottom_depth = float(f"{current_depth + layer.thickness:.1f}")
        f.write(f"{n_mesh}  0.0  {bottom_depth:.1f}\n")
        alpha_s = getattr(layer, 'shear_attenuation', 0.0)
        f.write(f"  {top_depth:.1f} {layer.sound_speed:.6f} "
                f"{layer.shear_speed:.1f} {layer.density:.2f} "
                f"{layer.attenuation:.2f} {alpha_s:.2f} /\n")
        f.write(f"  {bottom_depth:.1f} {layer.sound_speed:.6f} "
                f"{layer.shear_speed:.1f} {layer.density:.2f} "
                f"{layer.attenuation:.2f} {alpha_s:.2f} /\n")

        current_depth = bottom_depth

    return current_depth


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
    halfspace_alpha_s_source: str = 'zero',
    emit_reflection_table_block: bool = True,
    c_low: Optional[float] = None,
    c_high: Optional[float] = None,
    rmax: Optional[float] = None,
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
    cp_bottom, cs_bottom, rho_bottom, alpha_bottom : float, optional
        Halfspace overrides; default to ``env.bottom`` values.
    filepath : Path, optional
        Path to the ENV file being written (needed for copying .brc files)
    verbose : bool, optional
        Print verbose output
    halfspace_depth : float, optional
        Depth used for the 'A' halfspace line. Defaults to ``env.depth``
        plus stacked layered-bottom thicknesses.
    halfspace_alpha_s_source : {'zero', 'env'}
        Trailing column of the 'A' halfspace line. ``'zero'`` (Kraken/Bounce
        family) emits a literal ``0.0`` for shear attenuation; ``'env'``
        (Scooter) emits ``env.bottom.shear_attenuation``.
    emit_reflection_table_block : bool
        When the bottom is type ``'F'`` (reflection-coefficient table):
        emit the cmin/cmax/RMax bounds line that Kraken/Bounce expect.
        Scooter writes those bounds via ``write_phase_speed_and_rmax``
        instead and so passes ``False``.
    c_low, c_high, rmax : float, required when
        ``emit_reflection_table_block`` is ``True`` AND the bottom is type
        ``'F'``. Phase-velocity sampling bounds (m/s) and angle-resolution
        range (m) for the model that reads the ``.brc`` table.
    """
    hs = env.halfspace_at_range(0.0)
    if bottom_type is None:
        bottom_type = parse_boundary_type(hs.acoustic_type)

    cp = cp_bottom if cp_bottom is not None else hs.sound_speed
    cs = cs_bottom if cs_bottom is not None else getattr(hs, 'shear_speed', 0.0)
    rho = rho_bottom if rho_bottom is not None else hs.density
    alpha = alpha_bottom if alpha_bottom is not None else hs.attenuation

    bottom_code = bottom_type.to_acoustics_toolbox_code()
    sigma = getattr(hs, 'roughness', 0.0)

    if len(env.bathymetry) > 1:
        f.write(f"'{bottom_code}~' {sigma:.1f}\n")
    else:
        f.write(f"'{bottom_code}' {sigma:.1f}\n")

    if bottom_code == 'F':
        if hs.reflection_file and filepath is not None:
            import shutil
            brc_source = Path(hs.reflection_file)
            if brc_source.exists():
                brc_dest = filepath.with_suffix('.brc')
                shutil.copy(brc_source, brc_dest)
                log_message('oalib_writer',
                            f"copied reflection file: {brc_source} -> {brc_dest}",
                            verbose=verbose)
            else:
                raise FileNotFoundError(
                    f"at_env_writer: reflection coefficient file not found: "
                    f"{hs.reflection_file}; "
                    f"generate it via BOUNCE or OASR first."
                )
        elif hs.reflection_file is None:
            raise ValueError(
                "at_env_writer: acoustic_type='file' requires reflection_file= "
                "on the bottom BoundaryProperties (path to a .brc file)."
            )

        if emit_reflection_table_block:
            if c_low is None or c_high is None or rmax is None:
                raise ValueError(
                    "write_bottom_section: acoustic_type='file' with "
                    "emit_reflection_table_block=True requires "
                    "c_low, c_high, rmax to be passed by the caller."
                )
            f.write(f"{float(c_low):.2f}  {float(c_high):.2f}\n")
            f.write(f"{float(m_to_km(rmax)):.2f}\n")

    elif bottom_code == 'A':  # Half-space
        if halfspace_depth is not None:
            z_bottom = halfspace_depth
        else:
            z_bottom = env.depth
            if env.has_layered_bottom():
                z_bottom = float(f"{z_bottom:.1f}")
                for layer in env.bottom.layers:
                    z_bottom = float(f"{z_bottom + layer.thickness:.1f}")
        if halfspace_alpha_s_source == 'env':
            alpha_s = getattr(hs, 'shear_attenuation', 0.0)
        else:
            alpha_s = 0.0
        f.write(f"  {z_bottom:.2f}  {cp:.2f}  {cs:.1f}  "
                f"{rho:.2f}  {alpha:.2f}  {alpha_s:.2f} /\n")


def write_source_depths(f: TextIO, source: Source) -> None:
    """Write the source-depth section of an Acoustics Toolbox ``.env`` file."""
    n_sources = len(source.depths)
    f.write(f"{n_sources}\n")
    depths_str = " ".join([f"{d:.6f}" for d in source.depths])
    f.write(f"{depths_str} /\n")


def write_receiver_depths(f: TextIO, receiver_or_depths) -> None:
    """Write the receiver-depth section of an Acoustics Toolbox ``.env`` file.

    Accepts either a ``Receiver`` instance or a 1-D depths array.
    """
    depths = (
        receiver_or_depths.depths if isinstance(receiver_or_depths, Receiver)
        else np.asarray(receiver_or_depths, dtype=float)
    )
    f.write(f"{len(depths)}\n")
    depths_str = " ".join([f"{d:.6f}" for d in depths])
    f.write(f"{depths_str} /\n")


def write_receiver_ranges(f: TextIO, receiver: Receiver) -> None:
    """Write the receiver-range section (ranges converted from m to km)."""
    n_rr = len(receiver.ranges)
    f.write(f"{n_rr}\n")
    ranges_str = " ".join([f"{r/1000.0:.6f}" for r in receiver.ranges])
    f.write(f"{ranges_str} /\n")


def write_multi_profile_env(
    filepath: Union[str, Path],
    segments: List[Tuple[float, 'Environment']],
    source: Source,
    receiver: Receiver,
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
    **kwargs
        n_mesh, roughness, c_low, c_high, rmax_m passed through.
        TopOpt position 4 is taken from each segment env's ``absorption``
        field via :func:`write_header`.
    """
    n_mesh = kwargs.get('n_mesh', 0)
    roughness = kwargs.get('roughness', 0.0)
    c_low = kwargs.get('c_low', None)
    c_high = kwargs.get('c_high', None)
    rmax_m = kwargs.get('rmax_m', 100000.0)
    # If caller didn't specify, compute from max depth.
    if n_mesh <= 0:
        freq = float(source.frequencies[0])
        max_depth = max(seg.depth for _, seg in segments)
        n_mesh = max(500, int(max_depth * freq / 1500.0 * 20))

    # Determine max NMedia across all segments so every profile
    # can be padded to the same number of media (=> same NTotal).
    def _n_media(env_seg):
        n = 1
        if env_seg.has_layered_bottom():
            n += len(env_seg.bottom.layers)
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
        if env_seg.has_layered_bottom():
            for layer in env_seg.bottom.layers:
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

    interp_ssp = kwargs.get('interp_ssp', 'linear')

    with open(filepath, 'w') as f:
        for i, (_range_km, env_seg) in enumerate(segments):
            ssp_topopt = resolve_ssp_topopt(env_seg, interp_ssp)
            surface_obj = getattr(env_seg, 'surface', None)
            if surface_obj is not None:
                surface_type = parse_boundary_type(
                    surface_obj.acoustic_type
                )
            else:
                surface_type = BoundaryType.VACUUM
            bottom_type = parse_boundary_type(env_seg.halfspace_at_range(0.0).acoustic_type)

            n_media_this = _n_media(env_seg)
            n_media_write = max_n_media

            write_header(
                f, env_seg, source,
                ssp_topopt=ssp_topopt,
                surface_type=surface_type,
                n_media_override=n_media_write,
            )
            write_absorption_block(f, env_seg)

            # --- Water column (medium 1) ---
            write_ssp_section(
                f, env_seg, env_seg.depth,
                n_mesh=n_mesh,
                roughness=roughness
            )

            # --- Sediment layers (media 2..n_media_this) ---
            # Collect real layers with their depths, then write
            # them together with any needed extensions.
            # ``halfspace_at_range`` digs into ``LayeredBottom.halfspace`` so
            # a per-segment LayeredBottom (from RDLB → to_profile) still
            # exposes a flat halfspace for the padding-layer fields below.
            hs = env_seg.halfspace_at_range(0.0)
            seafloor = float(f"{env_seg.depth:.1f}")
            current_depth = seafloor
            real_layers = []

            if env_seg.has_layered_bottom():
                for layer in env_seg.bottom.layers:
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

            write_bottom_section(
                f, env_seg, bottom_type=bottom_type,
                filepath=filepath,
                verbose=kwargs.get('verbose', False),
                halfspace_depth=hs_depth,
                emit_reflection_table_block=False,
            )

            write_phase_speed_and_rmax(
                f, env_seg,
                rmax_m=rmax_m,
                c_low=c_low, c_high=c_high,
            )

            write_source_depths(f, source)
            write_receiver_depths(f, receiver)


def write_fieldflp(
    filepath: Union[str, Path],
    option: str,
    pos: Dict[str, Any],
    title: str = "",
    M_limit: int = 999999,
    n_profiles: int = 1,
    profile_ranges_km: Any = None,
) -> None:
    """
    Write field parameters file (.flp) for FIELD/FIELDS programs.

    Parameters
    ----------
    filepath : str or Path
        Output file path (extension .flp added if missing)
    option : str
        4-character option string for field.exe. Column semantics per AT
        ``field.f90:70-99`` and ``ReadModes.f90:315-324``:

        - Pos 1 (source type):
          'R' = cylindrical point source, 'X' = line source (Cartesian),
          'S' = scaled-cylindrical point source.
        - Pos 2 (coupling, for NProf > 1):
          'C' = coupled modes, 'A' = adiabatic.
        - Pos 3: either ``'*'`` to apply a ``.sbp`` source beam pattern
          or ``' '`` for omnidirectional. ``field.exe`` (``field.f90:83-90``)
          only accepts ``{' ', 'O', '*'}`` through this writer; elastic
          component selectors (``'P'``/``'H'``/``'V'``/``'T'``/``'N'``)
          are not reachable from uacpy.
        - Pos 4 (summation): 'C' = coherent, 'I' = incoherent.
    pos : dict
        Position dictionary with:
        - 's': dict with 'z' (source depths in m)
        - 'r': dict with 'z' (receiver depths in m), 'r' (ranges in m)
    title : str, optional
        Title for the file (default: empty)
    M_limit : int, optional
        Maximum number of modes to include (default: 999999 = all)
    n_profiles : int, optional
        Number of range profiles (default: 1 for range-independent).
        For range-dependent, set > 1 and provide profile_ranges_km.
    profile_ranges_km : array-like, optional
        Profile boundary ranges in km. Required when n_profiles > 1.
        First value must be 0.0. Length must equal n_profiles.

    Notes
    -----
    File format (.flp):
    - Line 1: Title
    - Line 2: Option (quoted, 4 chars)
    - Line 3: MLimit
    - Line 4: NProf (number of profiles)
    - Line 5: rProf (profile ranges in km)
    - Lines 6+: Receiver ranges, source depths, receiver depths, range offsets

    For range-dependent (NProf > 1), field.exe reads modes for each profile
    from a single .mod file produced by kraken.exe with multi-profile .env.

    See Also
    --------
    read_flp : Read field parameters file
    write_field3dflp : Write 3D field parameters
    """
    filepath = Path(filepath)
    if filepath.suffix != ".flp":
        filepath = filepath.with_suffix(".flp")

    r_ranges = m_to_km(pos["r"]["r"])
    s_depths = pos["s"]["z"]
    r_depths = pos["r"]["z"]

    # Validate profile parameters
    if n_profiles > 1:
        if profile_ranges_km is None:
            raise ValueError("profile_ranges_km required when n_profiles > 1")
        profile_ranges_km = np.asarray(profile_ranges_km, dtype=float)
        if len(profile_ranges_km) != n_profiles:
            raise ValueError(
                f"profile_ranges_km length ({len(profile_ranges_km)}) "
                f"must equal n_profiles ({n_profiles})"
            )
        if abs(profile_ranges_km[0]) > 1e-9:
            raise ValueError("First profile range must be 0.0 km")

    with open(filepath, "w") as f:
        f.write(f"'{title}' ! Title \n")

        # Option
        f.write(f"'{option:4s}'  ! Option \n")

        # Mode limit
        f.write(f"{M_limit}   ! Mlimit (number of modes to include) \n")

        # Profile info
        f.write(f"{n_profiles}        ! NProf  \n")
        if n_profiles == 1:
            f.write("0.0 /    ! rProf (km) \n")
        else:
            for r in profile_ranges_km:
                f.write(f"    {r:6f}  ")
            f.write("/ \t ! rProf (km) \n")

        # Receiver ranges
        f.write(f"{len(r_ranges):5d} \t \t \t \t ! NRr \n")
        if len(r_ranges) > 2 and equally_spaced(r_ranges):
            f.write(f"    {r_ranges[0]:6f}  {r_ranges[-1]:6f} ")
        else:
            for r in r_ranges:
                f.write(f"    {r:6f}  ")
        f.write("/ \t ! Rr(1)  ... (km) \n")

        # Source depths
        f.write(f"{len(s_depths):5d} \t \t \t \t ! NSz \n")
        if len(s_depths) > 2 and equally_spaced(s_depths):
            f.write(f"    {s_depths[0]:6f}  {s_depths[-1]:6f} ")
        else:
            for z in s_depths:
                f.write(f"    {z:6f}  ")
        f.write("/ \t ! Sz(1)  ... (m) \n")

        # Receiver depths
        f.write(f"{len(r_depths):5d} \t \t \t \t ! NRz \n")
        if len(r_depths) > 2 and equally_spaced(r_depths):
            f.write(f"    {r_depths[0]:6f}  {r_depths[-1]:6f} ")
        else:
            for z in r_depths:
                f.write(f"    {z:6f}  ")
        f.write("/ \t ! Rz(1)  ... (m) \n")

        # Receiver range offsets (array tilt) - default to zeros for every
        # receiver. AT's field.f90 enforces ``NRro == NRz`` (see
        # KrakenField/field.f90:147), so we keep the count = NRz. The
        # sentinel ``/`` terminator paired with a single explicit value
        # lets AT's SubTab routine replicate it across the full vector
        # (see misc/subtabulate.f90 — when x(3) is left at its -999.9
        # default, the vector is filled by repeating x(1)). This matches
        # the canonical AT examples MunkK.flp / DickinsK_rd.flp.
        f.write(f"{len(r_depths):5d} \t \t \t \t ! NRro \n")
        f.write("    0.0 /    \t \t \t \t ! Rro(1)  ... (m) \n")


def write_field3dflp(
    filepath: Union[str, Path],
    option: str,
    pos: Dict[str, Any],
    bathy: Dict[str, Any],
    mod_file_pattern: str = "'{}'",
    title: str = "",
    M_limit: int = 999999,
) -> None:
    """
    Write 3D field parameters file (.flp) for FIELD3D program.

    This creates a more complex .flp file for 3D acoustic field computation
    that includes bathymetry nodes, elements, and mode file references.

    Parameters
    ----------
    filepath : str or Path
        Output file path (extension .flp added if missing)
    option : str
        Option string (e.g., 'STDFM' for standard field mode)
    pos : dict
        Position dictionary with:
        - 's': dict with 'x', 'y', 'z' (source coords in m, m, m)
        - 'r': dict with 'z' (receiver depths in m), 'r' (ranges in m),
                         'theta' (bearings in degrees)
        - 'Nsx', 'Nsy': Number of source x,y points

        x/y/r values are converted to km on write to match the
        Bellhop3D ``.flp`` format. The public API stays in metres for
        symmetry with ``write_fieldflp`` and the 2-D sibling.
    bathy : dict
        Bathymetry dictionary with:
        - 'X': ndarray - X coordinates in m, shape (nx,) — converted
          to km on write.
        - 'Y': ndarray - Y coordinates in m, shape (ny,) — converted
          to km on write.
        - 'depth': ndarray - Depths in m, shape (ny, nx)
    mod_file_pattern : str, optional
        Pattern for mode file names. Can include format specifiers.
        Default: "'{}'" (single quoted)
    title : str, optional
        Title for the file (default: empty)
    M_limit : int, optional
        Maximum number of modes to include (default: 999999)

    Notes
    -----
    3D field parameters files specify:
    - Source/receiver positions in 3D
    - Bathymetry node locations (x, y, z)
    - Triangular mesh elements
    - Mode file for each node

    This is used for range and azimuth dependent propagation modeling.

    The bathymetry is represented as a triangulated surface where each
    node has an associated mode file containing normal modes for that
    location.

    Examples
    --------
    >>> import numpy as np
    >>> from uacpy.io import write_field3dflp
    >>>
    >>> # Set up 3D positions
    >>> pos = {
    ...     's': {'x': np.array([0]), 'y': np.array([0]), 'z': np.array([50])},
    ...     'r': {
    ...         'z': np.linspace(0, 100, 11),
    ...         'r': np.linspace(0, 50, 51),
    ...         'theta': np.linspace(0, 360, 37)
    ...     },
    ...     'Nsx': 1,
    ...     'Nsy': 1
    ... }
    >>>
    >>> # Set up bathymetry grid
    >>> X = np.linspace(0, 100, 11)
    >>> Y = np.linspace(0, 100, 11)
    >>> depth = 100 * np.ones((11, 11))
    >>> bathy = {'X': X, 'Y': Y, 'depth': depth}
    >>>
    >>> # Write 3D field parameters
    >>> write_field3dflp('field3d.flp', 'STDFM', pos, bathy,
    ...                  mod_file_pattern="'mode_{:07.1f}_{:07.1f}'",
    ...                  title='3D Test')

    See Also
    --------
    read_flp3d : Read 3D field parameters
    write_fieldflp : Write 2D field parameters
    """
    filepath = Path(filepath)
    if filepath.suffix != ".flp":
        filepath = filepath.with_suffix(".flp")

    s_x = m_to_km(pos["s"]["x"])
    s_y = m_to_km(pos["s"]["y"])
    s_z = pos["s"]["z"]
    r_z = pos["r"]["z"]
    r_r = m_to_km(pos["r"]["r"])
    r_theta = pos["r"]["theta"]
    Nsx = pos.get("Nsx", len(s_x))
    Nsy = pos.get("Nsy", len(s_y))

    X = m_to_km(bathy["X"])
    Y = m_to_km(bathy["Y"])
    depth = bathy["depth"]
    nx = len(X)
    ny = len(Y)

    with open(filepath, "w") as f:
        # Header
        f.write(f"'{title}' ! Title\n")
        f.write(f"'{option}' \t ! OPT\n")
        f.write(f"{M_limit}   ! Mlimit (number of modes to include)\n")

        # Source x-coordinates
        f.write(f"{Nsx}                 ! Nsx\n")
        if Nsx > 2 and equally_spaced(s_x):
            f.write(f"{s_x[0]} {s_x[-1]}          /   ! Sx( 1 : Nsx ) (km)\n")
        else:
            for x in s_x:
                f.write(f"{x} ")
            f.write("/ ! Sx (km)\n")

        # Source y-coordinates
        f.write(f"{Nsy}                 ! Nsy\n")
        if Nsy > 2 and equally_spaced(s_y):
            f.write(f"{s_y[0]} {s_y[-1]}          /   ! Sy( 1 : Nsy ) (km)\n")
        else:
            for y in s_y:
                f.write(f"{y} ")
            f.write("/ ! Sy (km)\n")

        # Source depths
        f.write(f"{len(s_z):5d} \t \t \t \t ! NSD\n")
        if len(s_z) > 2 and equally_spaced(s_z):
            f.write(f"    {s_z[0]:6f}  {s_z[-1]:6f} ")
        else:
            for z in s_z:
                f.write(f"    {z:6f}  ")
        f.write("/ \t ! SD(1)  ... (m)\n")

        # Receiver depths
        f.write(f"{len(r_z):5d} \t \t \t \t ! NRD\n")
        if len(r_z) > 2 and equally_spaced(r_z):
            f.write(f"    {r_z[0]:6f}  {r_z[-1]:6f} ")
        else:
            for z in r_z:
                f.write(f"    {z:6f}  ")
        f.write("/ \t ! RD(1)  ... (m)\n")

        # Receiver ranges
        f.write(f"{len(r_r):5d} \t \t \t \t ! NRR\n")
        if len(r_r) > 2 and equally_spaced(r_r):
            f.write(f"    {r_r[0]:6f}  {r_r[-1]:6f} ")
        else:
            for r in r_r:
                f.write(f"    {r:6f}  ")
        f.write("/ \t ! RR(1)  ... (km)\n")

        # Receiver bearings
        f.write(f"{len(r_theta)}              \n")
        if len(r_theta) > 2 and equally_spaced(r_theta):
            f.write(f"{r_theta[0]:.1f} {r_theta[-1]:.1f} /")
        else:
            for theta in r_theta:
                f.write(f"{theta:.1f} ")
            f.write("/")
        f.write("        ! NTHETA THETA(1:NTHETA) (degrees)\n")

        # Nodes
        nnodes = nx * ny
        f.write(f"{nnodes:5d}\n")

        # Write node data (x, y, mode_file)
        for iy in range(ny):
            for ix in range(nx):
                x_coord = X[ix]
                y_coord = Y[iy]
                z_depth = depth[iy, ix]

                # Generate mode file name
                if z_depth > 0:
                    if "{}" in mod_file_pattern or "{:" in mod_file_pattern:
                        modfil = mod_file_pattern.format(x_coord, y_coord)
                    else:
                        modfil = mod_file_pattern
                else:
                    modfil = "'DUMMY'"

                f.write(f"{x_coord:8.2f} {y_coord:8.2f} {modfil}\n")

        # Elements (triangular mesh)
        nelts = 2 * (nx - 1) * (ny - 1)
        f.write(f"{nelts:5d}\n")

        inode = 0
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                # Two triangles per grid cell
                f.write(f"{inode:5d} {inode + 1:5d} {inode + nx:5d}\n")
                f.write(f"{inode + 1:5d} {inode + nx:5d} {inode + nx + 1:5d}\n")
                inode += 1
            inode += 1
